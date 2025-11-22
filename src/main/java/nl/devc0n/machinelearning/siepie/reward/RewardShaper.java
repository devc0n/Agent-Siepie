package nl.devc0n.machinelearning.siepie.reward;

import nl.devc0n.machinelearning.siepie.model.Episode;
import nl.devc0n.machinelearning.siepie.model.GameStep;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.awt.image.BufferedImage;

public class RewardShaper {
    private static final float SURVIVAL_BONUS = 0.01f;
    private static final float DEATH_PENALTY = -10.0f;
    private static final int TERMINAL_WINDOW = 3;

    public void applyRewards(Episode episode) {
        markTerminalStates(episode);
        for (var step : episode.getSteps()) {
            step.setReward(calculateReward(step.isTerminal()));
        }
    }

    /**
     * After death detected, mark the last N frames as terminal states
     * This handles the delayed death detection problem
     */
    public void markTerminalStates(Episode episode) {
        if (episode.getSteps().isEmpty()) return;

        int episodeLength = episode.getSteps().size();
        int terminalStart = Math.max(0, episodeLength - TERMINAL_WINDOW);

        // Mark last N steps as terminal and apply death penalty
        for (int i = terminalStart; i < episodeLength; i++) {
            GameStep step = episode.getSteps().get(i);
            step.setTerminal(true);
            step.setReward(DEATH_PENALTY);

//            try{
//                BufferedImage[] frames = framesFromINDArray(step.getFrameStack());
//
//                for (int y = 0; y < frames.length; y++) {
//                    ImageIO.write(frames[y], "png", new File("step_"+i+"_frame_" + y + ".png"));
//                }
//            }catch (Exception e){
//                System.out.println("getting image went wrong");
//            }
        }
    }

    private BufferedImage[] framesFromINDArray(INDArray arr) {
        int K = (int) arr.size(0);
        int H = (int) arr.size(1);
        int W = (int) arr.size(2);

        BufferedImage[] images = new BufferedImage[K];

        for (int k = 0; k < K; k++) {
            BufferedImage img = new BufferedImage(W, H, BufferedImage.TYPE_BYTE_GRAY);

            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    // convert back to 0–255 pixel value
                    double v = arr.getDouble(k, y, x) * 255.0;
                    int gray = (int) v & 0xFF;
                    int rgb = (gray << 16) | (gray << 8) | gray;
                    img.setRGB(x, y, rgb);
                }
            }
            images[k] = img;
        }
        return images;
    }

    private float calculateReward(int currentStep, boolean gameOver, int episodeSteps) {
        if (gameOver) {
            // Proportionele penalty: hoe eerder dood, hoe erger
            float survivalRatio = (float) episodeSteps / 500.0f; // 500 = target
            return -50.0f * (1.0f - survivalRatio);
            // Dood op step 100: -40
            // Dood op step 300: -20
            // Dood op step 450: -5
        }

        // Quadratische groei voor late game
        if (currentStep > 300) {
            return 5.0f + (currentStep - 300) * 0.05f; // Extra bonus late game
        }

        return 1.0f + (currentStep * 0.01f);
    }

    private float calculateReward(boolean gameOver) {
        if (gameOver) {
            return -1.0f;  // Simpele death penalty
        }
        return 0.1f;  // Elke step = +0.1 (zoals origineel)
    }
}
