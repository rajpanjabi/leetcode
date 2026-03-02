
The Observer Pattern is a behavioral design pattern that defines a one-to-many dependency between objects so that when one object (the subject) changes its state, all its dependents (called observers) are notified and updated automatically.


Let's say we want to create a youtube playlist system, where we can add videos. Whenever a new video is uploaded, all the subscribers of the channel should be notified. Subscribers can be of different types like EmailSubscribers, MobileSubscribers, and so on..

So at ground level, we need a Subject interface, concrete class YoutubeSubject, a Subscriber interface, concrete implementations of Subscriber (Mobile, Email).
