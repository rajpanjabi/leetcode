package structural.Proxy;

public class Main {

    public static void main(String[] args) {
        ProxyVideoDownloader proxy = new ProxyVideoDownloader();
        System.out.println("User 1 downloading GOT");
        proxy.downloadVideo("Game of Thrones");
        System.out.println("User 2 downloading F1");
        proxy.downloadVideo("F1");
        System.out.println("User 3 downloading GOT");
        proxy.downloadVideo("Game of Thrones");

    }
    
}
