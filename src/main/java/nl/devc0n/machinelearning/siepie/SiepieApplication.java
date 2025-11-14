package nl.devc0n.machinelearning.siepie;

import nl.devc0n.machinelearning.siepie.game.GameRunnerService;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class SiepieApplication {

    public SiepieApplication(GameRunnerService gameRunnerService) {
    }

    public static void main(String[] args) throws  Exception {
		SpringApplication.run(SiepieApplication.class, args);
		GameRunnerService gameRunnerService = new GameRunnerService();
		gameRunnerService.playLoop(1);
	}
}
