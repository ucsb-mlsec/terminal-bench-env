const initialState = {
  items: [],
  total: 0,
  lastUpdated: null
};

export default function cartReducer(state = initialState, action) {
  switch (action.type) {
    case 'ADD_ITEM': {
      const existingItemIndex = state.items.findIndex(
        item => item.id === action.payload.id
      );

      if (existingItemIndex !== -1) {
        // MUTATION 1: Directly mutating existing item's quantity
        state.items[existingItemIndex].quantity += action.payload.quantity;
      } else {
        // MUTATION 2: Using push() which mutates the array
        state.items.push({
          id: action.payload.id,
          name: action.payload.name,
          price: action.payload.price,
          quantity: action.payload.quantity
        });
      }

      // MUTATION 3: Directly mutating state.total
      state.total = state.items.reduce(
        (sum, item) => sum + item.price * item.quantity,
        0
      );

      // MUTATION 4: Directly mutating state.lastUpdated
      state.lastUpdated = new Date().toISOString();

      return state;
    }

    case 'REMOVE_ITEM': {
      const itemIndex = state.items.findIndex(
        item => item.id === action.payload
      );

      if (itemIndex !== -1) {
        // MUTATION 5: Using splice() which mutates the array
        state.items.splice(itemIndex, 1);
      }

      // MUTATION 6: Directly mutating state.total
      state.total = state.items.reduce(
        (sum, item) => sum + item.price * item.quantity,
        0
      );

      // MUTATION 7: Directly mutating state.lastUpdated
      state.lastUpdated = new Date().toISOString();

      return state;
    }

    case 'UPDATE_QUANTITY': {
      const item = state.items.find(item => item.id === action.payload.id);

      if (item) {
        // MUTATION 8: Directly mutating item's quantity property
        item.quantity = action.payload.quantity;
      }

      // MUTATION 9: Directly mutating state.total
      state.total = state.items.reduce(
        (sum, item) => sum + item.price * item.quantity,
        0
      );

      // MUTATION 10: Directly mutating state.lastUpdated
      state.lastUpdated = new Date().toISOString();

      return state;
    }

    default:
      return state;
  }
}