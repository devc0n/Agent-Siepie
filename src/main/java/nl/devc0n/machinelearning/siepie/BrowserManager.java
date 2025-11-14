package nl.devc0n.machinelearning.siepie;

import org.openqa.selenium.Dimension;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;

import java.util.Map;

public class BrowserManager {

    private ChromeDriver driver;

    public void startBrowser(int instanceCount) {
        ChromeOptions options = new ChromeOptions();
        options.addArguments("--mute-audio"); // Mute audio
        options.addArguments("--headless");
        driver = new ChromeDriver(options);
        driver.manage().window().setSize(new Dimension(400, 900));
        driver.executeCdpCommand("Page.enable", Map.of());
        var position = driver.manage().window().getPosition();
        if (instanceCount > 1) {
            position.x = position.x + (516 * (instanceCount - 1));
        }

        driver.manage().window().setPosition(position);
        driver.get("https://sinterklaasspel.hema.nl");
    }

    public ChromeDriver getDriver() {
        return driver;
    }

    public void closeBrowser() {
        if (driver != null) {
            driver.quit();
        }
    }
}
