package nl.devc0n.machinelearning.siepie.game;

import nl.devc0n.machinelearning.siepie.BrowserManager;
import nl.devc0n.machinelearning.siepie.ai.*;
import org.nd4j.linalg.api.ndarray.INDArray;
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
import java.io.IOException;
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
        Map<String, Integer> clip = Map.of(
                "x", 125,
                "y", 230,
                "width", 250,
                "height", 300,
                "scale", 1
        );

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
            long t0 = System.currentTimeMillis();
            stepNumber++;

            BufferedImage[] inputFrames = frameStack.asArray();
            INDArray state = SupervisedTrainer.imageStackToINDArray(inputFrames);


            if (isGameOver(driver)) {

                var totalScore = scoreDetection(driver, gameHandler);
                screenshot = takeScreenshot(driver, clip);
                resized = resize(screenshot, 84, 84);
                gray = toGray(resized);
                frameStack.push(gray);

                BufferedImage[] finalFrames = frameStack.asArray();
                INDArray finalState = SupervisedTrainer.imageStackToINDArray(finalFrames);

                replayBuffer.add(new Transition(state, 99, totalScore, finalState, true));
                //todo: train




                episodeNumber++;
                stepNumber = 0;
                gameHandler.performAction(99);
                replayBuffer.clear();
                continue;
            }


            INDArray out = net.output(state);
            int action = Nd4j.argMax(out, 1).getInt(0);
            LOG.debug("action number {} performed", action);
            gameHandler.performAction(action);


            screenshot = takeScreenshot(driver, clip);
            resized = resize(screenshot, 84, 84);
            gray = toGray(resized);
            frameStack.push(gray);

            BufferedImage[] nextFrames = frameStack.asArray();
            INDArray nextState = SupervisedTrainer.imageStackToINDArray(nextFrames);
            replayBuffer.add(new Transition(state, action, 0, nextState, false));


            long dt = System.currentTimeMillis() - t0;
//            LOG.info("inference loop took {} ms", dt);
            if (dt > 200) {
//                LOG.warn("timeloop took longer than 200ms, it was {}ms", dt);
            }
            Thread.sleep(Math.max(0, 200 - dt));
        }
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
        var start = System.currentTimeMillis();
        Map<String, Object> result = driver.executeCdpCommand("Page.captureScreenshot", Map.of("clip", clip));
        String base64 = result.get("data").toString();
        byte[] decoded = Base64.getDecoder().decode(base64);
        var screenshot = ImageIO.read(new ByteArrayInputStream(decoded));
//        Files.write(Path.of("cdp-screenshot.png"), decoded);
//        LOG.info("Screenshot took {}ms", System.currentTimeMillis() - start);
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
