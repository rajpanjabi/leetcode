package src.behavioural.Strategy;

public class NearestDriverStrategy implements MatchingStrategy{
    @Override
    public void match(){
        System.out.println("Matching using NearestDriverStrategy");
    }
}
