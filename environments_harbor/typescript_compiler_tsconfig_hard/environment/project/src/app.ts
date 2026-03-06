import { formatDate } from '@shared/utils';
import { UserConfig } from '@shared/types';

interface UserProfile {
  id: string;
  name: string;
  email: string;
  createdAt: Date;
  config: UserConfig;
}

interface FormattedUserData {
  userId: string;
  userName: string;
  userEmail: string;
  registrationDate: string;
  settings: UserConfig;
}

export function processUserProfile(profile: UserProfile): FormattedUserData {
  return {
    userId: profile.id,
    userName: profile.name,
    userEmail: profile.email,
    registrationDate: formatDate(profile.createdAt),
    settings: profile.config
  };
}

export function initializeApp(config: UserConfig): void {
  console.log('Initializing application with config:', config);
  const now = new Date();
  console.log('Application started at:', formatDate(now));
}