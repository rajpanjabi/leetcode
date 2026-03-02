package src.behavioural.Iterator;

import java.util.List;

class YoutubePlaylistIterator implements PlaylistIterator {
    private List<Video> videos;
    private int position;

    public YoutubePlaylistIterator(List<Video> videos) {
        this.videos = videos;
        this.position = 0;
    }
    @Override
    public boolean hasNext(){
        return position<videos.size();
    }
    @Override
    public Video next(){
        return hasNext() ? videos.get(position++) : null;
    }
}
