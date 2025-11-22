package nl.devc0n.machinelearning.siepie;

import lombok.extern.slf4j.Slf4j;
import nl.devc0n.machinelearning.siepie.model.Action;
import org.openqa.selenium.By;
import org.openqa.selenium.Dimension;
import org.openqa.selenium.Keys;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.interactions.Actions;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.time.Duration;
import java.util.Base64;
import java.util.Map;

@Slf4j
public class BrowserManager {

    private final Map<String, Object> params = Map.of(
            "clip", Map.of(
                    "x", 75,
                    "y", 230,
                    "width", 350,
                    "height", 350,
                    "scale", 1),
            "format", "jpeg",
            "quality", 80);
    private ChromeDriver driver;
    private WebDriverWait wait;
    private Actions actions;

    public void startBrowser() {
        ChromeOptions options = new ChromeOptions();
        options.addArguments("--mute-audio"); // Mute audio

        // Optional: Run headless (no UI) for faster execution
        options.addArguments("--headless");

        driver = new ChromeDriver(options);
        driver.manage().window().setSize(new Dimension(400, 900));
        driver.executeCdpCommand("Page.enable", Map.of());
        var position = driver.manage().window().getPosition();

        driver.manage().window().setPosition(position);
        driver.get("https://sinterklaasspel.hema.nl");

        this.wait = new WebDriverWait(driver, Duration.ofSeconds(60));

        var allowCookiesButton = wait.until(ExpectedConditions.presenceOfElementLocated(By.id("CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll")));
        allowCookiesButton.click();
        log.debug("Allow cookies button was clicked");

        var playButton = wait.until(ExpectedConditions.presenceOfElementLocated(By.className("launch-button")));
        playButton.click();
        log.debug("Play button was clicked");

        var closeButton = wait.until(ExpectedConditions.presenceOfElementLocated(By.className("close")));
        closeButton.click();
        log.debug("Close button was clicked");

        actions = new Actions(driver);
    }

    public void performAction(Action action) {
        var start = System.currentTimeMillis();
        switch (action) {
            case UP -> actions.sendKeys(Keys.ARROW_UP).perform();
            case LEFT -> actions.sendKeys(Keys.ARROW_LEFT).perform();
            case RIGHT -> actions.sendKeys(Keys.ARROW_RIGHT).perform();
            case DOWN -> actions.sendKeys(Keys.ARROW_DOWN).perform();
            case NOTHING -> {
            }
        }
        log.debug("Performing action took: {}ms", System.currentTimeMillis() - start);
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
        var start = System.currentTimeMillis();
        Map<String, Object> result =
                driver.executeCdpCommand("Page.captureScreenshot", params);

        String base64 = result.get("data").toString();
        byte[] decoded = Base64.getDecoder().decode(base64);
        var screenshot = ImageIO.read(new ByteArrayInputStream(decoded));

        var output = preProcess(screenshot, 84, 84);

        log.debug("Screenshot capture duration: {}", System.currentTimeMillis() - start);
        return output;
    }

    private BufferedImage preProcess(BufferedImage src, int targetW, int targetH) throws IOException {
        BufferedImage out = new BufferedImage(targetW, targetH, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = out.createGraphics();

        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);

        g.drawImage(src, 0, 0, targetW, targetH, null);
        g.dispose();

//        File outputfile = new File("src/main/resources/screenshots/" + System.currentTimeMillis() + "-proc.png");
//        ImageIO.write(out, "png", outputfile);
        return out;
    }

}
