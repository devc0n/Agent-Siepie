package nl.devc0n.machinelearning.siepie.parallel;

import nl.devc0n.machinelearning.siepie.BrowserManager;
import nl.devc0n.machinelearning.siepie.agent.AgentSiepie;
import nl.devc0n.machinelearning.siepie.memory.ReplayBuffer;
import nl.devc0n.machinelearning.siepie.model.Action;
import nl.devc0n.machinelearning.siepie.model.Episode;
import nl.devc0n.machinelearning.siepie.model.FrameStack;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static nl.devc0n.machinelearning.siepie.SiepieApplication.saveFrameStackMontage;

public class ParallelTraining {

    private final AgentSiepie agent;
    private final ReplayBuffer sharedBuffer;
    private final int numBrowsers;
    private final ExecutorService executorService;
    private final AtomicBoolean running;

    public ParallelTraining(int numBrowsers) {
        this.numBrowsers = numBrowsers;
        this.sharedBuffer = new ReplayBuffer(1000);
        this.agent = new AgentSiepie(sharedBuffer);
        this.executorService = Executors.newFixedThreadPool(numBrowsers);
        this.running = new AtomicBoolean(true);
    }

    /**
     * Start parallel training with multiple browsers
     */
    public void start() {
        System.out.println("Starting parallel training with " + numBrowsers + " browsers");

        // Start browser worker threads
        for (int i = 0; i < numBrowsers; i++) {
            final int browserIndex = i;
            executorService.submit(() -> runBrowserWorker(browserIndex));
        }

        // Monitor progress in main thread
        monitorProgress();
    }

    /**
     * Browser worker - runs episodes and collects experience
     */
    private void runBrowserWorker(int browserIndex) {
        System.out.println("Browser worker " + browserIndex + " started");

        BrowserManager browserManager = new BrowserManager();
        browserManager.startBrowser();
        // Use different ports/profiles to run multiple Chrome instances
        // WebDriver driver = createDriver(9515 + browserIndex);

        int episodesCompleted = 0;

        while (running.get()) {
            try {
                // Run one episode
                Episode episode = new Episode();
                agent.startEpisode();

                var frameStack = new FrameStack(4, browserManager.takeScreenshot());
                INDArray state = frameStack.toINDArray();
                boolean died = false;

                while (!died && running.get()) {
                    // Select action
                    Action action = agent.selectAction(state);
                    browserManager.performAction(action);

                    frameStack.push(browserManager.takeScreenshot());
                    INDArray nextFrameStack = frameStack.toINDArray();


                    died = browserManager.detectDeathScreen();
                    if (!died) {
                        // Record step (training happens automatically inside)
                        agent.recordStep(episode, state, action, nextFrameStack);
                    }

                    state = nextFrameStack;
                }

                if (died) {
                    saveFrameStackMontage(state, "S:\\Development\\siepie\\screenshots\\montage_ep42_step100.png");
                    int finalScore = browserManager.extractFinalScore();
                    agent.endEpisode(episode, finalScore, 0);

                    browserManager.restartGame();
                    episodesCompleted++;

                    // Save checkpoint periodically
                    if (episodesCompleted % 50 == 0) {
                        saveCheckpoint(browserIndex, episodesCompleted);
                    }
                }

            } catch (Exception e) {
                System.err.println("Browser worker " + browserIndex + " error: " + e.getMessage());
                e.printStackTrace();
                // Continue running despite errors
            }
        }

        System.out.println("Browser worker " + browserIndex + " stopped after " +
                episodesCompleted + " episodes");
    }

    /**
     * Monitor and display training progress
     */
    private void monitorProgress() {
        long lastSteps = 0;
        long lastTime = System.currentTimeMillis();

        while (running.get()) {
            try {
                Thread.sleep(10000); // Update every 10 seconds

                long currentSteps = agent.getTotalSteps();
                long currentTime = System.currentTimeMillis();

                double stepsPerSecond = (currentSteps - lastSteps) /
                        ((currentTime - lastTime) / 1000.0);

                System.out.printf(
                        "=== Progress === Episodes: %d | Total Steps: %d | Buffer: %d episodes | " +
                                "Epsilon: %.3f | Steps/sec: %.1f%n",
                        agent.getEpisodeCount(),
                        currentSteps,
                        sharedBuffer.getEpisodeCount(),
                        agent.getEpsilon(),
                        stepsPerSecond
                );

                lastSteps = currentSteps;
                lastTime = currentTime;

            } catch (InterruptedException e) {
                break;
            }
        }
    }

    /**
     * Stop all browser workers and save final model
     */
    public void stop() {
        System.out.println("Stopping parallel training...");
        running.set(false);

        executorService.shutdown();
        try {
            if (!executorService.awaitTermination(30, TimeUnit.SECONDS)) {
                executorService.shutdownNow();
            }
        } catch (InterruptedException e) {
            executorService.shutdownNow();
        }

        // Save final model
        try {
            agent.save("models/final_model.zip");
            System.out.println("Final model saved");
        } catch (Exception e) {
            System.err.println("Failed to save final model: " + e.getMessage());
        }
    }

    private void saveCheckpoint(int browserIndex, int episodesCompleted) {
        try {
            String filename = String.format("models/checkpoint_browser%d_ep%d.zip",
                    browserIndex, episodesCompleted);
            agent.save(filename);
            System.out.println("Checkpoint saved: " + filename);
        } catch (Exception e) {
            System.err.println("Failed to save checkpoint: " + e.getMessage());
        }
    }
}
