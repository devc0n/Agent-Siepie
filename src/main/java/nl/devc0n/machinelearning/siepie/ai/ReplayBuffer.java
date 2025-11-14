package nl.devc0n.machinelearning.siepie.ai;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ReplayBuffer {
    private final List<Transition> buffer = new ArrayList<>();
    private final int capacity = 200000;

    public void add(Transition t) {
        if (buffer.size() >= capacity) {
            buffer.remove(0);
        }
        buffer.add(t);
    }

    public List<Transition> sample(int batchSize) {
        Collections.shuffle(buffer);
        return buffer.subList(0, batchSize);
    }

    public void clear(){
        this.buffer.clear();
    }
}

