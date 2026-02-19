#!/usr/bin/env python3
"""
Event Notification System with Memory Leak

This module implements an event dispatcher system that allows components
to subscribe to events and receive notifications. The current implementation
has a memory leak caused by circular references between subscribers and
the dispatcher.
"""

import weakref
from typing import Dict, List, Callable, Any
from collections import defaultdict


# Event type constants
USER_LOGIN = 'user_login'
USER_LOGOUT = 'user_logout'
USER_ACTIVITY = 'user_activity'
SYSTEM_EVENT = 'system_event'


class EventDispatcher:
    """
    Central event dispatcher that manages subscriptions and dispatches events.
    
    This class maintains a registry of event subscribers and notifies them
    when events occur. WARNING: Current implementation has memory leak issues
    due to circular references with subscriber objects.
    """
    
    def __init__(self):
        """Initialize the event dispatcher with empty subscription lists."""
        # Dictionary mapping event types to lists of callback functions
        self._subscribers = defaultdict(list)
        
        # Statistics tracking
        self._stats = {
            'total_dispatches': 0,
            'total_subscriptions': 0,
            'total_unsubscriptions': 0
        }
        
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Subscribe a callback to a specific event type.
        
        Args:
            event_type: The type of event to subscribe to
            callback: The function to call when the event occurs
        """
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)
            self._stats['total_subscriptions'] += 1
            
    def unsubscribe(self, event_type: str, callback: Callable) -> bool:
        """
        Unsubscribe a callback from a specific event type.
        
        Args:
            event_type: The type of event to unsubscribe from
            callback: The callback function to remove
            
        Returns:
            True if the callback was found and removed, False otherwise
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
                self._stats['total_unsubscriptions'] += 1
                
                # Clean up empty event type entries
                if not self._subscribers[event_type]:
                    del self._subscribers[event_type]
                    
                return True
            except ValueError:
                pass
        return False
    
    def dispatch(self, event_type: str, data: Any = None) -> int:
        """
        Dispatch an event to all subscribed callbacks.
        
        Args:
            event_type: The type of event to dispatch
            data: Optional data to pass to callbacks
            
        Returns:
            The number of callbacks that were notified
        """
        count = 0
        if event_type in self._subscribers:
            # Create a copy to avoid issues if callbacks modify subscriptions
            callbacks = self._subscribers[event_type][:]
            
            for callback in callbacks:
                try:
                    callback(data)
                    count += 1
                except Exception as e:
                    print(f"Error in callback {callback}: {e}")
                    
        self._stats['total_dispatches'] += 1
        return count
    
    def get_subscriber_count(self, event_type: str = None) -> int:
        """
        Get the number of subscribers for an event type or total.
        
        Args:
            event_type: Optional event type to query. If None, returns total.
            
        Returns:
            The number of subscribers
        """
        if event_type:
            return len(self._subscribers.get(event_type, []))
        return sum(len(callbacks) for callbacks in self._subscribers.values())
    
    def get_stats(self) -> Dict[str, int]:
        """Get dispatcher statistics."""
        return self._stats.copy()
    
    def clear_all(self) -> None:
        """Clear all subscriptions. Use with caution."""
        self._subscribers.clear()


class Subscriber:
    """
    A subscriber that can register for and receive event notifications.
    
    This class demonstrates the memory leak issue: it holds a reference to
    the dispatcher, and the dispatcher holds references to its bound methods,
    creating a circular reference that prevents garbage collection.
    """
    
    def __init__(self, name: str, dispatcher: EventDispatcher):
        """
        Initialize a subscriber with a name and dispatcher reference.
        
        Args:
            name: Identifier for this subscriber
            dispatcher: The EventDispatcher to subscribe to
        """
        self.name = name
        # MEMORY LEAK: This creates a circular reference
        # Subscriber holds dispatcher, dispatcher holds bound methods,
        # bound methods hold subscriber instance
        self.dispatcher = dispatcher
        
        # Track which events we're subscribed to
        self._subscriptions = []
        
        # Stats for this subscriber
        self.notifications_received = 0
        
    def subscribe_to_all(self) -> None:
        """Subscribe to all available event types."""
        self.subscribe_to_login()
        self.subscribe_to_logout()
        self.subscribe_to_activity()
        
    def subscribe_to_login(self) -> None:
        """Subscribe to user login events."""
        # MEMORY LEAK: self.on_user_login is a bound method that holds
        # a reference to self, creating a circular reference
        self.dispatcher.subscribe(USER_LOGIN, self.on_user_login)
        self._subscriptions.append((USER_LOGIN, self.on_user_login))
        
    def subscribe_to_logout(self) -> None:
        """Subscribe to user logout events."""
        self.dispatcher.subscribe(USER_LOGOUT, self.on_user_logout)
        self._subscriptions.append((USER_LOGOUT, self.on_user_logout))
        
    def subscribe_to_activity(self) -> None:
        """Subscribe to user activity events."""
        self.dispatcher.subscribe(USER_ACTIVITY, self.on_user_activity)
        self._subscriptions.append((USER_ACTIVITY, self.on_user_activity))
        
    def on_user_login(self, data: Any) -> None:
        """Handle user login event."""
        self.notifications_received += 1
        # Process login notification
        
    def on_user_logout(self, data: Any) -> None:
        """Handle user logout event."""
        self.notifications_received += 1
        # Process logout notification
        
    def on_user_activity(self, data: Any) -> None:
        """Handle user activity event."""
        self.notifications_received += 1
        # Process activity notification
        
    def unsubscribe_all(self) -> None:
        """
        Unsubscribe from all events.
        
        This attempts to clean up subscriptions, but the circular reference
        still prevents the Subscriber from being garbage collected because
        the dispatcher still holds references to the bound methods.
        """
        for event_type, callback in self._subscriptions:
            self.dispatcher.unsubscribe(event_type, callback)
        self._subscriptions.clear()
        
    def __del__(self):
        """Destructor - should be called when object is garbage collected."""
        # In a leaked scenario, this won't be called
        pass


class NotificationManager:
    """
    High-level manager for the notification system.
    
    Provides a convenient interface for creating subscribers and
    dispatching events.
    """
    
    def __init__(self):
        """Initialize the notification manager."""
        self.dispatcher = EventDispatcher()
        self.active_subscribers = {}
        
    def create_subscriber(self, name: str) -> Subscriber:
        """
        Create and register a new subscriber.
        
        Args:
            name: Identifier for the subscriber
            
        Returns:
            The newly created Subscriber instance
        """
        subscriber = Subscriber(name, self.dispatcher)
        self.active_subscribers[name] = subscriber
        return subscriber
        
    def remove_subscriber(self, name: str) -> bool:
        """
        Remove a subscriber by name.
        
        Args:
            name: The subscriber identifier
            
        Returns:
            True if removed, False if not found
        """
        if name in self.active_subscribers:
            subscriber = self.active_subscribers[name]
            subscriber.unsubscribe_all()
            del self.active_subscribers[name]
            return True
        return False
        
    def notify_login(self, user_data: Any) -> None:
        """Dispatch a login event."""
        self.dispatcher.dispatch(USER_LOGIN, user_data)
        
    def notify_logout(self, user_data: Any) -> None:
        """Dispatch a logout event."""
        self.dispatcher.dispatch(USER_LOGOUT, user_data)
        
    def notify_activity(self, activity_data: Any) -> None:
        """Dispatch an activity event."""
        self.dispatcher.dispatch(USER_ACTIVITY, activity_data)


if __name__ == '__main__':
    # Example usage
    manager = NotificationManager()
    
    # Create subscribers
    sub1 = manager.create_subscriber('user1')
    sub1.subscribe_to_all()
    
    # Dispatch events
    manager.notify_login({'user': 'alice'})
    manager.notify_activity({'action': 'click'})
    
    # Cleanup
    manager.remove_subscriber('user1')
    
    print("Event system initialized")