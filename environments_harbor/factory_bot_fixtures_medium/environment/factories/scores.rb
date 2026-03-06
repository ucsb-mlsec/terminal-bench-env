require 'factory_bot'

FactoryBot.define do
  factory :score do
    points { 1500 }
    game_session { nil }
  end
end