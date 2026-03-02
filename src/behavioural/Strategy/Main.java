package src.behavioural.Strategy;

public class Main {

    public static void main(String[] args) {
        RideMatchingService service = new RideMatchingService(new NearestDriverStrategy());
        service.riderMatch();
        service.setMatchingStrategy(new airportStrategy());
        service.riderMatch();
        
    }

    
}
