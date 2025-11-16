package nl.devc0n.machinelearning.siepie.game;

import nl.devc0n.machinelearning.siepie.BrowserManager;
import nl.devc0n.machinelearning.siepie.ai.FrameStack;
import nl.devc0n.machinelearning.siepie.ai.ModelFactory;
import nl.devc0n.machinelearning.siepie.ai.ReplayBuffer;
import nl.devc0n.machinelearning.siepie.ai.SupervisedTrainer;
import nl.devc0n.machinelearning.siepie.ai.Transition;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Base64;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@Service
public class GameRunnerService {

    private final String MODEL_PATH = "policyNetwork-2025-11-15-19.zip";

    Logger LOG = LoggerFactory.getLogger(GameRunnerService.class);
    private double epsilon = 1; //Exploring
//    private double epsilon = 0; //Doing

    public void playLoop(int numberOfInstances) throws Exception {
        try (ExecutorService executor = Executors.newFixedThreadPool(numberOfInstances)) {
            for (int count = 1; count <= numberOfInstances; count++) {
                int gameId = count; // Capture the game ID for the thread
                executor.submit(() -> {
                    try {
                        playGame(gameId);
                    } catch (Exception e) {
                        LOG.error(e.getMessage());
                        throw new RuntimeException(e);
                    }
                });
            }
        }
    }

    private void playGame(int gameId) throws Exception {
        BrowserManager browserManager = new BrowserManager();
        browserManager.startBrowser(gameId);
        var driver = browserManager.getDriver();
        GameHandler gameHandler = new GameHandler(driver);
        gameHandler.firstStart();

        // Capture cropped screenshot
        Map<String, Integer> clip =
                Map.of("x", 125, "y", 230, "width", 250, "height", 300, "scale", 1);

        var playing = true;
        var stepNumber = 0;
        var episodeNumber = 1;
        var screenshot = takeScreenshot(driver, clip);
        var resized = resize(screenshot, 84, 84);
        var gray = toGray(resized);
        var frameStack = new FrameStack(4, gray);

        MultiLayerNetwork network;

        if (Files.exists(Paths.get(MODEL_PATH))) {
            network = ModelSerializer.restoreMultiLayerNetwork(MODEL_PATH, true);
            LOG.info("Loaded network {} from disk", MODEL_PATH);
        } else {
            var modelFactory = new ModelFactory();
            network = modelFactory.createPolicyNetwork(4, 84, 84, 5);
            LOG.info("Created a new network");
        }

        var replayBuffer = new ReplayBuffer();
        while (playing) {

            if (isGameOver(driver)) {

                var totalScore = scoreDetection(driver, gameHandler);

                trainEpisode(network, replayBuffer, totalScore);

                if (episodeNumber % 50 == 0) {
                    saveNetwork(network);
                }

                episodeNumber++;
                stepNumber = 0;
                gameHandler.performAction(99);
                replayBuffer.clear();
                epsilon = Math.max(0.1, epsilon * 0.995);  // Decay epsilon (minimum 0.1)

                continue;
            }
            long t0 = System.currentTimeMillis();
            stepNumber++;

            // Take a screenshot and feed it to the network.
            screenshot = takeScreenshot(driver, clip);
            resized = resize(screenshot, 84, 84);
            gray = toGray(resized);
            frameStack.push(gray);
            BufferedImage[] inputFrames = frameStack.asArray();
            INDArray state = SupervisedTrainer.imageStackToINDArray(inputFrames);
            INDArray actionProbabilities = network.output(state);

            var action = selectActionWithEpsilonGreedy(network, state, epsilon);
//            int action = Nd4j.argMax(actionProbabilities, 1).getInt(0); //todo: softmax instead of argmax?

            // get the action with the highest success chance according to the network.
            LOG.debug("action number {} performed", action);
            gameHandler.performAction(action);

            replayBuffer.add(new Transition(state, action));

            long dt = System.currentTimeMillis() - t0;
            LOG.debug("inference loop took {} ms", dt);
            if (dt > 200) {
                LOG.debug("timeloop took longer than 200ms, it was {}ms", dt);
            }
            Thread.sleep(Math.max(0, 200 - dt));
        }
    }

    public int selectActionWithEpsilonGreedy(MultiLayerNetwork net, INDArray state,
            double epsilon) {
        // Get the action probabilities or Q-values from the network
        INDArray actionProbabilities = net.output(state);  // Shape: [batchSize, nActions]

        // If epsilon is greater than a random number, explore
        if (Math.random() < epsilon) {
            // Explore: choose a random action
            return (int) (Math.random() * actionProbabilities.size(1));  // Random action
        } else {
            // Exploit: choose the action with the highest value (greedy)
            return Nd4j.argMax(actionProbabilities, 1).getInt(0);  // Max action
        }
    }


    public void saveNetwork(MultiLayerNetwork net) {
        try {
            DateTimeFormatter formatter2 = DateTimeFormatter.ofPattern("yyy-MM-dd-HH-mm");
            String formattedString2 =
                    ZonedDateTime.now(ZoneId.of("Europe/Amsterdam")).format(formatter2);


            String filename = "src/main/resources/policyNetwork-" + formattedString2 + ".zip";
            File locationToSave = new File(filename);

            boolean saveUpdater = true; // includes optimizer state (Adam momentum etc.)
            ModelSerializer.writeModel(net, locationToSave, saveUpdater);

            System.out.println("Saved model: " + filename);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public void trainEpisode(MultiLayerNetwork network, ReplayBuffer replayBuffer,
            double finalScore) {
        var transitions = replayBuffer.getBuffer();
        int batchSize = transitions.size();
        int nActions = 5; // adjust as needed
        int channels = 4; // number of stacked frames
        int height = 84;
        int width = 84;

        if (batchSize == 0) {
            return; // nothing to train on
        }

        // Stack all states along the batch dimension
        INDArray[] stateArrays = transitions.stream()
                .map(Transition::state)      // each state shape: [1, channels, H, W]
                .toArray(INDArray[]::new);

        // Concatenate along axis 0 to get shape [batchSize, channels, H, W]
        INDArray states = Nd4j.concat(0, stateArrays);

        // Create labels (one-hot actions)
        INDArray labels = Nd4j.zeros(batchSize, nActions);
        for (int i = 0; i < batchSize; i++) {
            int action = transitions.get(i).action();
            labels.putScalar(new int[] {i, action}, 1.0);
        }

        // Mask must be shape [batchSize, 1]
        INDArray labelMask = Nd4j.ones(batchSize, 1).mul(finalScore);

        DataSet ds = new DataSet(states, labels);
        ds.setLabelsMaskArray(labelMask);
        network.fit(ds);
    }


    private boolean isGameOver(ChromeDriver driver) {
        try {
            driver.findElement(By.className("score-points-current"));
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private BufferedImage takeScreenshot(ChromeDriver driver, Map<String, Integer> clip)
            throws IOException {
        Map<String, Object> result =
                driver.executeCdpCommand("Page.captureScreenshot", Map.of("clip", clip));
        String base64 = result.get("data").toString();
        byte[] decoded = Base64.getDecoder().decode(base64);
        var screenshot = ImageIO.read(new ByteArrayInputStream(decoded));
//        Files.write(Path.of("cdp-screenshot.png"), decoded);
        return screenshot;
    }

    private double scoreDetection(WebDriver driver, GameHandler gameHandler)
            throws InterruptedException {
        Thread.sleep(5000);
        int score = Integer.parseInt(
                driver.findElement(By.className("score-points-current")).getText());
        LOG.info("Current score is {}", score);
        return score;
    }

    private BufferedImage resize(BufferedImage src, int targetW, int targetH) {
        BufferedImage out = new BufferedImage(targetW, targetH, BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D g = out.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                           RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(src, 0, 0, targetW, targetH, null);
        g.dispose();
        return out;
    }

    private BufferedImage toGray(BufferedImage src) {
        BufferedImage gray =
                new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics g = gray.getGraphics();
        g.drawImage(src, 0, 0, null);
        g.dispose();
        return gray;
    }


}
