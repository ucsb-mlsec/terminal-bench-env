require 'factory_bot'

FactoryBot.define do
  factory :game_session do
    session_id { 'duplicate_id' }
    player { nil }
  end
end