// Shared type definitions for the project

/**
 * Configuration interface for application settings
 */
export interface Config {
  name: string;
  port: number;
  enabled: boolean;
}

/**
 * User configuration interface
 */
export interface UserConfig {
  userId: string;
  username: string;
  email: string;
  settings: {
    theme: string;
    language: string;
  };
}

/**
 * Status type representing possible states
 */
export type Status = 'active' | 'inactive' | 'pending';

/**
 * Generic API response interface
 */
export interface ApiResponse<T> {
  data: T;
  status: number;
  message: string;
}