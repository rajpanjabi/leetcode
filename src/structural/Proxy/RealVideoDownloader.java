package structural.Proxy;

public class RealVideoDownloader implements VideoDownloader {
    @Override
    public String downloadVideo(String videoUrl){
        System.out.println("Downloading video");
        return "Video content from " + videoUrl;

    }
    
}
    
