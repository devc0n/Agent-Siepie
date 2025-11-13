package nl.devc0n.machinelearning.siepie.game;

import jakarta.annotation.PreDestroy;
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.springframework.stereotype.Service;

import java.time.Duration;

@Service
public class GameRunnerService {

    private WebDriver driver;

    public void startGame(String gameUrl){
        var chromeOptions = new ChromeOptions();
        // Optional: run headless to save resources
        // chromeOptions.addArguments("--headless=new");
        chromeOptions.addArguments("--disable-gpu");
        chromeOptions.addArguments("--window-size=1280,800");

        driver = new ChromeDriver(chromeOptions);
        driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(10));

        System.out.println("Opening game: " + gameUrl);
        driver.get(gameUrl);

        // Wait for the main game canvas to appear
        WebElement canvas = driver.findElement(By.cssSelector("canvas"));
        System.out.println("Game canvas found: " + (canvas != null));

        // You can later capture a screenshot or prepare to send keystrokes
    }

    public void takeScreenshot(String filename) {
        // TODO: Implement this later
    }

    @PreDestroy
    public void shutdown() {
        if (driver != null) {
            driver.quit();
        }
    }

}
