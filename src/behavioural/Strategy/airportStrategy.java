package src.behavioural.Strategy;

public class airportStrategy implements MatchingStrategy{
    
    @Override
    public void match(){
        System.out.println("Matching using Airport Strategy");
    }
}
