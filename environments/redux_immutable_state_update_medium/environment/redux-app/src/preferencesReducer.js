// Preferences Reducer for User Settings
// This reducer manages user preference state including theme, notifications, display options, and feature flags

const initialState = {
  theme: 'light',
  notifications: { email: true, push: false, sms: true },
  displayOptions: { compactView: false, showAvatars: true, itemsPerPage: 20 },
  featureFlags: { betaFeatures: false, experimentalUI: false, advancedMode: false }
};

const preferencesReducer = (state = initialState, action) => {
  switch (action.type) {
    case 'SET_THEME':
      // BUG: Direct state mutation - modifying state.theme directly
      state.theme = action.payload;
      return state;

    case 'UPDATE_NOTIFICATION_SETTINGS':
      // BUG: Direct state mutation - modifying nested notifications object directly
      state.notifications[action.key] = action.value;
      return state;

    case 'TOGGLE_COMPACT_VIEW':
      // BUG: Direct state mutation - modifying nested displayOptions object directly
      state.displayOptions.compactView = !state.displayOptions.compactView;
      return state;

    case 'SET_ITEMS_PER_PAGE':
      // BUG: Direct state mutation - modifying nested displayOptions property directly
      state.displayOptions.itemsPerPage = action.payload;
      return state;

    case 'TOGGLE_FEATURE_FLAG':
      // BUG: Direct state mutation - modifying nested featureFlags object directly
      state.featureFlags[action.flagName] = !state.featureFlags[action.flagName];
      return state;

    case 'UPDATE_DISPLAY_OPTIONS':
      // BUG: Direct state mutation - using Object.assign to mutate existing nested object
      Object.assign(state.displayOptions, action.payload);
      return state;

    case 'RESET_NOTIFICATIONS':
      // BUG: Direct state mutation - replacing nested object but returning original state reference
      state.notifications = { email: false, push: false, sms: false };
      return state;

    case 'SET_SHOW_AVATARS':
      // BUG: Direct state mutation - modifying nested displayOptions property directly
      state.displayOptions.showAvatars = action.payload;
      return state;

    default:
      return state;
  }
};

export default preferencesReducer;