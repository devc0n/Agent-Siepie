package nl.devc0n.machinelearning.siepie;

import lombok.extern.slf4j.Slf4j;
import nl.devc0n.machinelearning.siepie.agent.AgentSiepie;
import nl.devc0n.machinelearning.siepie.memory.MovingAverageTracker;
import nl.devc0n.machinelearning.siepie.memory.ReplayBuffer;
import nl.devc0n.machinelearning.siepie.model.Action;
import nl.devc0n.machinelearning.siepie.model.Episode;
import nl.devc0n.machinelearning.siepie.model.FrameStack;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

@SpringBootApplication
@Slf4j
public class SiepieApplication {

    public static void main(String[] args) throws Exception {
        SpringApplication.run(SiepieApplication.class, args);

        ReplayBuffer sharedReplayBuffer = new ReplayBuffer(1000);
        AgentSiepie agent = new AgentSiepie(sharedReplayBuffer);
        MovingAverageTracker movingAvg20 = new MovingAverageTracker(20);
        MovingAverageTracker movingAvg100 = new MovingAverageTracker(100);

//       Optional: Load previous checkpoint
        try {
            agent.load("fresh.zip");
            log.info("Starting from checkpoint");
        } catch (IOException e) {
            log.info("Starting fresh training");
        }

        BrowserManager browserManager = new BrowserManager();
        browserManager.startBrowser();

        int episodeNum = 0;
        while (true) {
            episodeNum++;
            Episode episode = new Episode();
            agent.startEpisode();

            var frameStack = new FrameStack(4, browserManager.takeScreenshot());
            var state = frameStack.toINDArray();

            boolean died = false;

            var stepCount = 0;
            while (!died) {
                stepCount++;
                long start = System.nanoTime();
                // Select action
                Action action = agent.selectAction(state);

                browserManager.performAction(action);

                var screenshot = browserManager.takeScreenshot();
                frameStack.push(screenshot);

                INDArray nextState = frameStack.toINDArray();

                died = browserManager.detectDeathScreen(); // Your method

                if (!died) {
                    // Record the step
                    agent.recordStep(episode, state, action, nextState);
                }

                state = nextState;
                long remaining = 200 - (System.nanoTime() - start) / 1_000_000;
                if (remaining > 0) Thread.sleep(remaining);
            }

            int finalScore = browserManager.extractFinalScore();
            agent.endEpisode(episode, finalScore, episodeNum);

            // Update moving averages
            double avg20 = movingAvg20.addScore(stepCount);
            double avg100 = movingAvg100.addScore(stepCount);

            browserManager.restartGame();

            // Save checkpoint every 10 episodes
            if (episodeNum % 10 == 0 && episodeNum > 0) {
                try {
                    agent.save("checkpoint_" + episodeNum + ".zip");
                } catch (Exception e) {
                    e.printStackTrace();
                }
                log.info("==> Episode {} | MA-20: {} | MA-100: {}, Max steps: {}",
                        episodeNum, avg20, avg100, movingAvg100.getHiScore());
            }
        }
    }

    /**
     * Debug method: Convert INDArray frame stack back to images and save them
     * Useful for verifying what the network actually sees
     *
     * @param frameStack INDArray with shape [4, 84, 84] or [1, 4, 84, 84]
     * @param outputDir  Directory to save images (e.g., "debug_frames")
     * @param prefix     Filename prefix (e.g., "step_42")
     */
    public static void saveFrameStackAsImages(INDArray frameStack, String outputDir, String prefix) {
        try {
            // Handle both [4, 84, 84] and [1, 4, 84, 84] shapes
            INDArray frames = frameStack;
            if (frameStack.rank() == 4) {
                // Remove batch dimension: [1, 4, 84, 84] -> [4, 84, 84]
                frames = frameStack.get(NDArrayIndex.point(0), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all());
            }

            int K = (int) frames.size(0); // Number of frames (should be 4)
            int H = (int) frames.size(1); // Height (should be 84)
            int W = (int) frames.size(2); // Width (should be 84)

            // Create output directory if it doesn't exist
            File dir = new File(outputDir);
            if (!dir.exists()) {
                dir.mkdirs();
            }

            // Save each frame
            for (int k = 0; k < K; k++) {
                BufferedImage img = new BufferedImage(W, H, BufferedImage.TYPE_BYTE_GRAY);

                for (int y = 0; y < H; y++) {
                    for (int x = 0; x < W; x++) {
                        // Get normalized value [0, 1]
                        double normalizedValue = frames.getDouble(k, y, x);

                        // Convert back to [0, 255]
                        int gray = (int) Math.round(normalizedValue * 255.0);
                        gray = Math.max(0, Math.min(255, gray)); // Clamp to valid range

                        // Create grayscale RGB value
                        int rgb = (gray << 16) | (gray << 8) | gray;
                        img.setRGB(x, y, rgb);
                    }
                }

                // Save image
                String filename = String.format("%s/%s_frame_%d.png", outputDir, prefix, k);
                File outputFile = new File(filename);
                ImageIO.write(img, "png", outputFile);

                log.debug("Saved: " + filename);
            }

        } catch (Exception e) {
            System.err.println("Error saving frame stack: " + e.getMessage());
            e.printStackTrace();
        }
    }


    /**
     * Create a side-by-side comparison image of all 4 frames
     * Useful to see temporal progression at a glance
     */
    public static void saveFrameStackMontage(INDArray frameStack, String outputPath) {
        try {
            // Handle batch dimension
            INDArray frames = frameStack;
            if (frameStack.rank() == 4) {
                frames = frameStack.get(NDArrayIndex.point(0), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all());
            }

            int K = (int) frames.size(0);
            int H = (int) frames.size(1);
            int W = (int) frames.size(2);

            // Create montage: 2x2 grid of frames
            int rows = 2;
            int cols = 2;
            BufferedImage montage = new BufferedImage(W * cols, H * rows, BufferedImage.TYPE_BYTE_GRAY);
            Graphics2D g = montage.createGraphics();

            for (int k = 0; k < K; k++) {
                BufferedImage frame = new BufferedImage(W, H, BufferedImage.TYPE_BYTE_GRAY);

                for (int y = 0; y < H; y++) {
                    for (int x = 0; x < W; x++) {
                        double normalizedValue = frames.getDouble(k, y, x);
                        int gray = (int) Math.round(normalizedValue * 255.0);
                        gray = Math.max(0, Math.min(255, gray));
                        int rgb = (gray << 16) | (gray << 8) | gray;
                        frame.setRGB(x, y, rgb);
                    }
                }

                // Position in grid
                int row = k / cols;
                int col = k % cols;
                g.drawImage(frame, col * W, row * H, null);

                // Add label
                g.setColor(Color.WHITE);
                g.drawString("Frame " + k, col * W + 5, row * H + 15);
            }

            g.dispose();

            File outputFile = new File(outputPath);
            ImageIO.write(montage, "png", outputFile);
            log.debug("Saved montage: " + outputPath);

        } catch (Exception e) {
            System.err.println("Error saving montage: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
