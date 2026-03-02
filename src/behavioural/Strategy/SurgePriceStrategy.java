package src.behavioural.Strategy;

public class SurgePriceStrategy implements MatchingStrategy{
    
    @Override
    public void match(){
        System.out.println("Matching using Surge Price Strategy");
    }
}
