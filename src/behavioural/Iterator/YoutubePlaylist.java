package src.behavioural.Iterator;

import java.util.ArrayList;
import java.util.List;

class YoutubePlaylist implements Playlist{
    private List<Video> videos = new ArrayList<>();

    public void addVideo(Video video){
        videos.add(video);
    }
    
    @Override
    public PlaylistIterator createIterator(){
        return new YoutubePlaylistIterator(videos);
    }
}
