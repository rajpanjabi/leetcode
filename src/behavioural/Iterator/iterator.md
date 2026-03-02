
Iterator pattern is a behavioural design pattern that provides a way to access the elements of a collection sequentially without exposing the underlying representation.

That means, whether the underlying collection is an array, a list, a tree or something custom, we can use an iterator to traverse it in a consistent manner, one element at a time, without worrying about how the data is stored or managed internally.

For instance, let's say we're building a youtube playlist system. we want to store a list of videos and print their titles one by one.

```java
class Video {
    String title;

    public Video(String title) {
        this.title = title;
    }

    public String getTitle() {
        return title;
    }
}

class YouTubePlaylist {
    private List<Video> videos = new ArrayList<>();
       
    // Add a video to the playlist
    public void addVideo(Video video) {
        videos.add(video);
    }

    // Expose the video list
    public List<Video> getVideos() {
        return videos;
    }
}

// Client code
class Main {
    public static void main(String[] args) {
        YouTubePlaylist playlist = new YouTubePlaylist();
        playlist.addVideo(new Video("LLD Tutorial"));
        playlist.addVideo(new Video("System Design Basics"));

        // Loop through videos and print titles
        // this is not how it should work
        // This breaks encapsulation, as clients can access or even modify the internal collection outside the owning class. Also there is tight coupling between client and specific type of collection used.

        for (Video v : playlist.getVideos()) {
            System.out.println(v.getTitle());
        }
    }
}
```

Now using iterator pattern, we expose a youtubePlaylistIterator and playlistIterator