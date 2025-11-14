package nl.devc0n.machinelearning.siepie.ai;

import lombok.Getter;

import java.util.ArrayList;
import java.util.List;

@Getter
public class ReplayBuffer {
    private final List<Transition> buffer = new ArrayList<>();
    private final int capacity = 200000;

    public void add(Transition t) {
        if (buffer.size() >= capacity) {
            buffer.remove(0);
        }
        buffer.add(t);
    }

    public void clear(){
        this.buffer.clear();
    }
}

