const cartReducer = require('../src/reducers/cartReducer.js');

describe('cartReducer', () => {
  describe('Mutation Detection - ADD_ITEM', () => {
    test('does not mutate original state when adding first item', () => {
      const originalState = {
        items: [],
        total: 0,
        lastUpdated: null
      };
      const clonedState = JSON.parse(JSON.stringify(originalState));
      
      const action = {
        type: 'ADD_ITEM',
        payload: { id: 1, name: 'Product A', price: 29.99, quantity: 1 }
      };
      
      cartReducer(originalState, action);
      
      expect(originalState).toEqual(clonedState);
    });

    test('does not mutate original state when adding to existing cart', () => {
      const originalState = {
        items: [
          { id: 1, name: 'Product A', price: 29.99, quantity: 2 }
        ],
        total: 59.98,
        lastUpdated: '2024-01-15T10:30:00Z'
      };
      const clonedState = JSON.parse(JSON.stringify(originalState));
      
      const action = {
        type: 'ADD_ITEM',
        payload: { id: 2, name: 'Product B', price: 49.99, quantity: 1 }
      };
      
      cartReducer(originalState, action);
      
      expect(originalState).toEqual(clonedState);
    });

    test('does not mutate original state when adding duplicate item', () => {
      const originalState = {
        items: [
          { id: 1, name: 'Product A', price: 29.99, quantity: 2 }
        ],
        total: 59.98,
        lastUpdated: '2024-01-15T10:30:00Z'
      };
      const clonedState = JSON.parse(JSON.stringify(originalState));
      
      const action = {
        type: 'ADD_ITEM',
        payload: { id: 1, name: 'Product A', price: 29.99, quantity: 1 }
      };
      
      cartReducer(originalState, action);
      
      expect(originalState).toEqual(clonedState);
    });
  });

  describe('ADD_ITEM Functionality', () => {
    test('adds first item to empty cart', () => {
      const state = {
        items: [],
        total: 0,
        lastUpdated: null
      };
      
      const action = {
        type: 'ADD_ITEM',
        payload: { id: 1, name: 'Product A', price: 29.99, quantity: 2 }
      };
      
      const newState = cartReducer(state, action);
      
      expect(newState.items.length).toBe(1);
      expect(newState.items[0]).toEqual({ id: 1, name: 'Product A', price: 29.99, quantity: 2 });
      expect(newState.total).toBe(59.98);
      expect(newState.lastUpdated).not.toBeNull();
    });

    test('adds new item to cart with existing items', () => {
      const state = {
        items: [
          { id: 1, name: 'Product A', price: 29.99, quantity: 2 }
        ],
        total: 59.98,
        lastUpdated: '2024-01-15T10:30:00Z'
      };
      
      const action = {
        type: 'ADD_ITEM',
        payload: { id: 2, name: 'Product B', price: 49.99, quantity: 1 }
      };
      
      const newState = cartReducer(state, action);
      
      expect(newState.items.length).toBe(2);
      expect(newState.items[1]).toEqual({ id: 2, name: 'Product B', price: 49.99, quantity: 1 });
      expect(newState.total).toBe(109.97);
    });

    test('increments quantity when adding duplicate item', () => {
      const state = {
        items: [
          { id: 1, name: 'Product A', price: 29.99, quantity: 2 }
        ],
        total: 59.98,
        lastUpdated: '2024-01-15T10:30:00Z'
      };
      
      const action = {
        type: 'ADD_ITEM',
        payload: { id: 1, name: 'Product A', price: 29.99, quantity: 3 }
      };
      
      const newState = cartReducer(state, action);
      
      expect(newState.items.length).toBe(1);
      expect(newState.items[0].quantity).toBe(5);
      expect(newState.total).toBe(149.95);
    });
  });

  describe('Mutation Detection - REMOVE_ITEM', () => {
    test('does not mutate original state when removing item', () => {
      const originalState = {
        items: [
          { id: 1, name: 'Product A', price: 29.99, quantity: 2 },
          { id: 2, name: 'Product B', price: 49.99, quantity: 1 }
        ],
        total: 109.97,
        lastUpdated: '2024-01-15T10:30:00Z'
      };
      const clonedState = JSON.parse(JSON.stringify(originalState));
      
      const action = {
        type: 'REMOVE_ITEM',
        payload: { id: 1 }
      };
      
      cartReducer(originalState, action);
      
      expect(originalState).toEqual(clonedState);
    });

    test('does not mutate original state items array', () => {
      const originalState = {
        items: [
          { id: 1, name: 'Product A', price: 29.99, quantity: 2 }
        ],
        total: 59.98,
        lastUpdated: '2024-01-15T10:30:00Z'
      };
      const originalItemsLength = originalState.items.length;
      
      const action = {
        type: 'REMOVE_ITEM',
        payload: { id: 1 }
      };
      
      cartReducer(originalState, action);
      
      expect(originalState.items.length).toBe(originalItemsLength);
    });
  });

  describe('REMOVE_ITEM Functionality', () => {
    test('removes existing item from cart', () => {
      const state = {
        items: [
          { id: 1, name: 'Product A', price: 29.99, quantity: 2 },
          { id: 2, name: 'Product B', price: 49.99, quantity: 1 }
        ],
        total: 109.97,
        lastUpdated: '2024-01-15T10:30:00Z'
      };
      
      const action = {
        type: 'REMOVE_ITEM',
        payload: { id: 1 }
      };
      
      const newState = cartReducer(state, action);
      
      expect(newState.items.length).toBe(1);
      expect(newState.items[0].id).toBe(2);
      expect(newState.total).toBe(49.99);
    });

    test('removes last item from cart', () => {
      const state = {
        items: [
          { id: 1, name: 'Product A', price: 29.99, quantity: 2 }
        ],
        total: 59.98,
        lastUpdated: '2024-01-15T10:30:00Z'
      };
      
      const action = {
        type: 'REMOVE_ITEM',
        payload: { id: 1 }
      };
      
      const newState = cartReducer(state, action);
      
      expect(newState.items.length).toBe(0);
      expect(newState.total).toBe(0);
    });

    test('returns unchanged state when removing non-existent item', () => {
      const state = {
        items: [
          { id: 1, name: 'Product A', price: 29.99, quantity: 2 }
        ],
        total: 59.98,
        lastUpdated: '2024-01-15T10:30:00Z'
      };
      
      const action = {
        type: 'REMOVE_ITEM',
        payload: { id: 999 }
      };
      
      const newState = cartReducer(state, action);
      
      expect(newState.items.length).toBe(1);
      expect(newState.total).toBe(59.98);
    });
  });

  describe('Mutation Detection - UPDATE_QUANTITY', () => {
    test('does not mutate original state when updating quantity', () => {
      const originalState = {
        items: [
          { id: 1, name: 'Product A', price: 29.99, quantity: 2 }
        ],
        total: 59.98,
        lastUpdated: '2024-01-15T10:30:00Z'
      };
      const clonedState = JSON.parse(JSON.stringify(originalState));
      
      const action = {
        type: 'UPDATE_QUANTITY',
        payload: { id: 1, quantity: 5 }
      };
      
      cartReducer(originalState, action);
      
      expect(originalState).toEqual(clonedState);
    });

    test('does not mutate original item objects', () => {
      const originalState = {
        items: [
          { id: 1, name: 'Product A', price: 29.99, quantity: 2 },
          { id: 2, name: 'Product B', price: 49.99, quantity: 1 }
        ],
        total: 109.97,
        lastUpdated: '2024-01-15T10:30:00Z'
      };
      const originalQuantity = originalState.items[0].quantity;
      
      const action = {
        type: 'UPDATE_QUANTITY',
        payload: { id: 1, quantity: 5 }
      };
      
      cartReducer(originalState, action);
      
      expect(originalState.items[0].quantity).toBe(originalQuantity);
    });
  });

  describe('UPDATE_QUANTITY Functionality', () => {
    test('updates quantity of existing item', () => {
      const state = {
        items: [
          { id: 1, name: 'Product A', price: 29.99, quantity: 2 }
        ],
        total: 59.98,
        lastUpdated: '2024-01-15T10:30:00Z'
      };
      
      const action = {
        type: 'UPDATE_QUANTITY',
        payload: { id: 1, quantity: 5 }
      };
      
      const newState = cartReducer(state, action);
      
      expect(newState.items[0].quantity).toBe(5);
      expect(newState.total).toBe(149.95);
    });

    test('updates quantity and recalculates total correctly', () => {
      const state = {
        items: [
          { id: 1, name: 'Product A', price: 29.99, quantity: 2 },
          { id: 2, name: 'Product B', price: 49.99, quantity: 1 }
        ],
        total: 109.97,
        lastUpdated: '2024-01-15T10:30:00Z'
      };
      
      const action = {
        type: 'UPDATE_QUANTITY',
        payload: { id: 2, quantity: 3 }
      };
      
      const newState = cartReducer(state, action);
      
      expect(newState.items[1].quantity).toBe(3);
      expect(newState.total).toBe(209.95);
    });

    test('does not modify other items when updating quantity', () => {
      const state = {
        items: [
          { id: 1, name: 'Product A', price: 29.99, quantity: 2 },
          { id: 2, name: 'Product B', price: 49.99, quantity: 1 }
        ],
        total: 109.97,
        lastUpdated: '2024-01-15T10:30:00Z'
      };
      
      const action = {
        type: 'UPDATE_QUANTITY',
        payload: { id: 1, quantity: 5 }
      };
      
      const newState = cartReducer(state, action);
      
      expect(newState.items[1].quantity).toBe(1);
      expect(newState.items[1]).toEqual({ id: 2, name: 'Product B', price: 49.99, quantity: 1 });
    });
  });

  describe('Edge Cases', () => {
    test('returns initial state for undefined state', () => {
      const action = { type: 'UNKNOWN_ACTION' };
      const newState = cartReducer(undefined, action);
      
      expect(newState).toEqual({
        items: [],
        total: 0,
        lastUpdated: null
      });
    });

    test('returns unchanged state for unknown action type', () => {
      const state = {
        items: [
          { id: 1, name: 'Product A', price: 29.99, quantity: 2 }
        ],
        total: 59.98,
        lastUpdated: '2024-01-15T10:30:00Z'
      };
      
      const action = { type: 'UNKNOWN_ACTION' };
      const newState = cartReducer(state, action);
      
      expect(newState).toEqual(state);
    });

    test('handles zero quantity update', () => {
      const state = {
        items: [
          { id: 1, name: 'Product A', price: 29.99, quantity: 2 }
        ],
        total: 59.98,
        lastUpdated: '2024-01-15T10:30:00Z'
      };
      
      const action = {
        type: 'UPDATE_QUANTITY',
        payload: { id: 1, quantity: 0 }
      };
      
      const newState = cartReducer(state, action);
      
      expect(newState.items[0].quantity).toBe(0);
      expect(newState.total).toBe(0);
    });
  });
});