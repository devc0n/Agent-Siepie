package nl.devc0n.machinelearning.siepie;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.awt.AWTException;
import java.awt.Rectangle;
import java.awt.Robot;
import java.awt.image.BufferedImage;

@Service
public class ScreenshotUtil {

    private final Logger LOG = LoggerFactory.getLogger(ScreenshotUtil.class);

    public BufferedImage takeScreenshot(org.openqa.selenium.Rectangle rect) throws AWTException {
        var width = 250;
        var height = 300;

        var x = rect.getX() + 18 + 125;
        var y = rect.getY() + 150 + 230;

        Rectangle screenRect = new Rectangle(x, y, width, height);

        return new Robot().createScreenCapture(screenRect);
    }

}
