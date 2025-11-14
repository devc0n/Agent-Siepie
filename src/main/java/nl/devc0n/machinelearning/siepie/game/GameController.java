package nl.devc0n.machinelearning.siepie.game;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/game")
public class GameController {

    private final GameRunnerService gameRunnerService;

    public GameController(GameRunnerService gameRunnerService) {
        this.gameRunnerService = gameRunnerService;
    }

    @PostMapping("/start")
    public String start(@RequestParam String url, @RequestParam boolean save, @RequestParam int numberOfInstances) throws Exception {
        gameRunnerService.playLoop(1);
        return "Game started at " + url;
    }
}
