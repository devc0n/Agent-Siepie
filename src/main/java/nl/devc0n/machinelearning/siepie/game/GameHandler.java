package nl.devc0n.machinelearning.siepie.game;

import org.apache.juli.logging.LogFactory;
import org.openqa.selenium.By;
import org.openqa.selenium.Keys;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

public class GameHandler {

    private final WebDriver driver;
    private final WebDriverWait wait;
    private final Map<Integer, Integer> actionMap = new HashMap<>();

    private final Logger LOG = LoggerFactory.getLogger(GameHandler.class);

    public GameHandler(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, Duration.ofSeconds(60));
    }

    public void performAction(int action) throws InterruptedException {
        actionMap.put(action, actionMap.getOrDefault(action, 0) + 1);

        switch (action) {
            case 0:
                return;
            case 1:
                driver.findElement(By.tagName("body")).sendKeys(Keys.ARROW_UP);
                return;
            case 2:
                driver.findElement(By.tagName("body")).sendKeys(Keys.ARROW_LEFT);
                return;
            case 3:
                driver.findElement(By.tagName("body")).sendKeys(Keys.ARROW_RIGHT);
                return;
            case 4:
                driver.findElement(By.tagName("body")).sendKeys(Keys.ARROW_DOWN);
                return;
            case 99:
                var startGameButton = wait.until(ExpectedConditions.presenceOfElementLocated(By.className("icon-only")));
                startGameButton.click();
                Thread.sleep(4000);
                return;
            default:
                System.out.println("Unknown action: " + action);
                return;
        }
    }

    public void firstStart() {
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

}
