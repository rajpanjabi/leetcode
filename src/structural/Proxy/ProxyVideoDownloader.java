package structural.Proxy;

import java.util.HashMap;
import java.util.Map;

public class ProxyVideoDownloader implements VideoDownloader {
    // Here we cache the video and check if it exists/was downloaded already
    private RealVideoDownloader videoDownloader;
    private Map<String,String> cache;

    public ProxyVideoDownloader(){
        this.videoDownloader=new RealVideoDownloader();
        this.cache= new HashMap<>();
    }

    @Override
    public String downloadVideo(String videoURL){
        // first check if video already exists in cache
        if (cache.containsKey(videoURL)){
            System.out.println("Video exists in cache");
            return cache.get(videoURL);
        }
        else{
            System.out.println("Cache miss. Downloading...");
            String content = videoDownloader.downloadVideo(videoURL);
            cache.put(videoURL,content);
            return content;
        }
    }
}
   
