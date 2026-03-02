package src.behavioural.Strategy;

public class RideMatchingService {
    private MatchingStrategy strategy;

    public RideMatchingService(MatchingStrategy strategy){
        this.strategy=strategy;
    }
    public void setMatchingStrategy(MatchingStrategy strategy){
        this.strategy=strategy;
    }
    public void riderMatch(){
        strategy.match();
    }

    
}
