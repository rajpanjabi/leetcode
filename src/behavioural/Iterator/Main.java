package src.behavioural.Iterator;

public class Main {
    public static void main(String[] args) {    
        // create a playlist
        YoutubePlaylist playlist = new YoutubePlaylist();
        // add videos
        playlist.addVideo(new Video("DSA"));
        playlist.addVideo(new Video("LLD"));
        // get hold of iterator
        PlaylistIterator iterator=playlist.createIterator();

        while (iterator.hasNext()){
            System.out.println(iterator.next().getTitle());
        }

}

}