package nl.devc0n.machinelearning.siepie;

import nl.devc0n.machinelearning.siepie.agent.AgentSiepie;
import nl.devc0n.machinelearning.siepie.memory.ReplayBuffer;
import nl.devc0n.machinelearning.siepie.model.Action;
import nl.devc0n.machinelearning.siepie.model.Episode;
import nl.devc0n.machinelearning.siepie.model.FrameStack;
import nl.devc0n.machinelearning.siepie.parallel.ParallelTraining;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.IOException;

@SpringBootApplication
public class SiepieApplication {

    public static void main(String[] args) throws Exception {
        SpringApplication.run(SiepieApplication.class, args);

        int numBrowsers = 4;
        System.out.println("Starting parallel training with " + numBrowsers + " browsers");

        // Create and start parallel training
        ParallelTraining training = new ParallelTraining(numBrowsers);

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\nShutdown signal received...");
            training.stop();
        }));

        // Start training
        training.start();



        ReplayBuffer sharedReplayBuffer = new ReplayBuffer(1000);
        AgentSiepie agent = new AgentSiepie(sharedReplayBuffer);

//         Optional: Load previous checkpoint
         try {
             agent.load("src/main/resources/models/checkpoint_530.zip");
         } catch (IOException e) {
             System.out.println("Starting fresh training");
         }

        int maxEpisodes = 1000;

        BrowserManager browserManager = new BrowserManager();
        browserManager.startBrowser();

        System.out.printf(
                "%-10s %-10s %-10s %-10s %-15s%n",
                "Episode", "Score", "Steps", "Epsilon", "BufferSize"
        );
        for (int episodeNum = 0; episodeNum < maxEpisodes; episodeNum++) {
            Episode episode = new Episode();
            agent.startEpisode();

            var frameStack = new FrameStack(4, browserManager.takeScreenshot());
            var state = frameStack.toINDArray();

            boolean died = false;

            while (!died) {
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
            agent.endEpisode(episode, finalScore);

            browserManager.restartGame();

            // Save checkpoint every 10 episodes
            if (episodeNum % 10 == 0 && episodeNum > 0) {
                try {
                    agent.save("src/main/resources/models/checkpoint_" + episodeNum + ".zip");
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
