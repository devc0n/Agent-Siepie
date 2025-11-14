package nl.devc0n.machinelearning.siepie.game;

import nl.devc0n.machinelearning.siepie.BrowserManager;
import nl.devc0n.machinelearning.siepie.ai.*;
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
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Base64;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@Service
public class GameRunnerService {


    Logger LOG = LoggerFactory.getLogger(GameRunnerService.class);

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
        Map<String, Integer> clip = Map.of("x", 125, "y", 230, "width", 250, "height", 300, "scale", 1);

        var playing = true;
        var stepNumber = 0;
        var episodeNumber = 1;
        var screenshot = takeScreenshot(driver, clip);
        var resized = resize(screenshot, 84, 84);
        var gray = toGray(resized);
        var frameStack = new FrameStack(4, gray);
        var modelFactory = new ModelFactory();
        var net = modelFactory.createPolicyNetwork(4, 84, 84, 5);
        var replayBuffer = new ReplayBuffer();
        while (playing) {

            if (isGameOver(driver)) {

                var totalScore = scoreDetection(driver, gameHandler);

                trainEpisode(net, replayBuffer, totalScore);

                if (episodeNumber % 50 == 0) {
                    saveNetwork(net);
                }

                episodeNumber++;
                stepNumber = 0;
                gameHandler.performAction(99);
                replayBuffer.clear();
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
            INDArray out = net.output(state);

            // get the action with the highest success chance according to the network.
            int action = Nd4j.argMax(out, 1).getInt(0); //todo: softmax instead of argmax?
            LOG.debug("action number {} performed", action);
            gameHandler.performAction(action);


            replayBuffer.add(new Transition(state, action));


//            long dt = System.currentTimeMillis() - t0;
//            LOG.info("inference loop took {} ms", dt);
//            if (dt > 200) {
//                LOG.warn("timeloop took longer than 200ms, it was {}ms", dt);
//            }
//            Thread.sleep(Math.max(0, 200 - dt));
        }
    }

    public void saveNetwork(MultiLayerNetwork net) {
        try {
            DateTimeFormatter formatter2 = DateTimeFormatter.ofPattern("yyy-MM-dd-HH:mm");
            String formattedString2 = ZonedDateTime.now(ZoneId.of("Europe/Amsterdam")).format(formatter2);

            String filename = "policyNetwork-" + formattedString2 + ".zip";
            File locationToSave = new File(filename);

            boolean saveUpdater = true; // includes optimizer state (Adam momentum etc.)
            ModelSerializer.writeModel(net, locationToSave, saveUpdater);

            System.out.println("Saved model: " + filename);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public void trainEpisode(MultiLayerNetwork network, ReplayBuffer replayBuffer, double finalScore) {
        var transitions = replayBuffer.getBuffer();
        int batchSize = transitions.size();
        int nActions = 5; // adjust as needed
        int channels = 4; // number of stacked frames
        int height = 84;
        int width = 84;

        if (batchSize == 0) return; // nothing to train on

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
            labels.putScalar(new int[]{i, action}, 1.0);
        }

        // Create label mask to weight the loss by the final score (REINFORCE)
        INDArray labelMask = Nd4j.ones(batchSize).mul(finalScore);

        // Build dataset
        DataSet ds = new DataSet(states, labels);
        ds.setLabelsMaskArray(labelMask);

        // Fit network
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

    private BufferedImage takeScreenshot(ChromeDriver driver, Map<String, Integer> clip) throws IOException {
        Map<String, Object> result = driver.executeCdpCommand("Page.captureScreenshot", Map.of("clip", clip));
        String base64 = result.get("data").toString();
        byte[] decoded = Base64.getDecoder().decode(base64);
        var screenshot = ImageIO.read(new ByteArrayInputStream(decoded));
//        Files.write(Path.of("cdp-screenshot.png"), decoded);
        return screenshot;
    }

    private double scoreDetection(WebDriver driver, GameHandler gameHandler) throws InterruptedException {
        Thread.sleep(5000);
        int score = Integer.parseInt(driver.findElement(By.className("score-points-current")).getText());
        LOG.info("Current score is {}", score);
        return score;
    }

    private BufferedImage resize(BufferedImage src, int targetW, int targetH) {
        BufferedImage out = new BufferedImage(targetW, targetH, BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D g = out.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(src, 0, 0, targetW, targetH, null);
        g.dispose();
        return out;
    }

    private BufferedImage toGray(BufferedImage src) {
        BufferedImage gray = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics g = gray.getGraphics();
        g.drawImage(src, 0, 0, null);
        g.dispose();
        return gray;
    }


}
