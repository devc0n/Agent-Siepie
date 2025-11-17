package nl.devc0n.machinelearning.siepie.model;

public enum Action {
    NOTHING(0), LEFT(1), RIGHT(2), UP(3), DOWN(4);

    public final int index;

    Action(int index) {
        this.index = index;
    }

    public static Action fromIndex(int i) {
        for (Action a : values()) {
            if (a.index == i) return a;
        }
        return NOTHING;
    }

    public static int getNumActions() {
        return values().length;
    }
}
