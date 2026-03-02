package src.behavioural.Observer;

 class MobileSubscriber implements Subscriber{
    private String username;
    public MobileSubscriber(String username){
        this.username=username;
    }
    @Override
    public void update(String title){
        System.out.println("Sending mobile notification to "+username +" for new video: "+ title);
    }
}
