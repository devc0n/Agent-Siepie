package nl.devc0n.machinelearning.siepie;

import nl.devc0n.machinelearning.siepie.model.Action;
import org.openqa.selenium.By;
import org.openqa.selenium.Dimension;
import org.openqa.selenium.Keys;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.util.Base64;
import java.util.Map;

public class BrowserManager {

    private ChromeDriver driver;
    private WebDriverWait wait;

    private final Map<String, Integer> clip = Map.of("x", 100, "y", 230, "width", 300, "height", 300, "scale", 1);

    private final Logger LOG = LoggerFactory.getLogger(BrowserManager.class);


    public void startBrowser() {
        ChromeOptions options = new ChromeOptions();
        options.addArguments("--mute-audio"); // Mute audio
//        options.addArguments("--headless");
        driver = new ChromeDriver(options);
        driver.manage().window().setSize(new Dimension(400, 900));
        driver.executeCdpCommand("Page.enable", Map.of());
        var position = driver.manage().window().getPosition();

        driver.manage().window().setPosition(position);
        driver.get("https://sinterklaasspel.hema.nl");

        this.wait = new WebDriverWait(driver, Duration.ofSeconds(60));

        var allowCookiesButton = wait.until(ExpectedConditions.presenceOfElementLocated(By.id("CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll")));
        allowCookiesButton.click();
        LOG.debug("Allow cookies button was clicked");

        var playButton = wait.until(ExpectedConditions.presenceOfElementLocated(By.className("launch-button")));
        playButton.click();
        LOG.debug("Play button was clicked");

        var closeButton = wait.until(ExpectedConditions.presenceOfElementLocated(By.className("close")));
        closeButton.click();
        LOG.debug("Close button was clicked");
    }

    public void performAction(Action action) throws InterruptedException {
        switch (action) {
            case NOTHING:
                return;
            case UP:
                driver.findElement(By.tagName("body")).sendKeys(Keys.ARROW_UP);
                return;
            case LEFT:
                driver.findElement(By.tagName("body")).sendKeys(Keys.ARROW_LEFT);
                return;
            case RIGHT:
                driver.findElement(By.tagName("body")).sendKeys(Keys.ARROW_RIGHT);
                return;
            case DOWN:
                driver.findElement(By.tagName("body")).sendKeys(Keys.ARROW_DOWN);
                return;
            default:
        }
    }

    public void restartGame() throws InterruptedException {
        var startGameButton = wait.until(ExpectedConditions.presenceOfElementLocated(By.className("icon-only")));
        startGameButton.click();
        Thread.sleep(4000);
    }

    public boolean detectDeathScreen() {
        try {
            driver.findElement(By.className("score-points-current"));
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    public int extractFinalScore() throws InterruptedException {
        var score = 0;
        var changing = true;
        while (changing) {
            Thread.sleep(500);
            int newScore = Integer.parseInt(driver.findElement(By.className("score-points-current")).getText());
            if (score == newScore && score > 0) {
                changing = false;
            } else {
                score = newScore;
            }
        }
        return score;
    }

    public void closeBrowser() {
        if (driver != null) {
            driver.quit();
        }
    }

    public BufferedImage takeScreenshot() throws IOException {
        Map<String, Object> result = driver.executeCdpCommand("Page.captureScreenshot", Map.of("clip", clip));
        String base64 = result.get("data").toString();
        byte[] decoded = Base64.getDecoder().decode(base64);
        var screenshot = ImageIO.read(new ByteArrayInputStream(decoded));
//        File outputfile = new File("src/main/resources/screenshots/" + System.currentTimeMillis() + "-original.png");
//        ImageIO.write(screenshot, "png", outputfile);
        return resizeAndGrayscale(screenshot, 84, 84, false);
    }

    private BufferedImage resizeAndGrayscale(BufferedImage src, int targetW, int targetH, boolean highQuality) throws IOException {
        //todo: see difference in timing with highQuality
        BufferedImage out = new BufferedImage(targetW, targetH, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = out.createGraphics();

        if (highQuality) {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                    RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        } else {
            // Faster but lower quality
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                    RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
        }

        g.drawImage(src, 0, 0, targetW, targetH, null);
        g.dispose();

//        File outputfile = new File("src/main/resources/screenshots/" + System.currentTimeMillis() + "-proc.png");
//        ImageIO.write(out, "png", outputfile);

        return out;
    }

}
