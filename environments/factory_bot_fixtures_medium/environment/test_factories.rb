# Test script to verify FactoryBot factories are working correctly
# This script attempts to use the factories to generate test data
# Currently, the factories are broken and will cause validation failures

require 'factory_bot'
require './models/player'
require './models/game_session'
require './models/score'

# Load factory definitions
require './factories/players'
require './factories/game_sessions'
require './factories/scores'

# Include FactoryBot methods to use build/create directly
include FactoryBot::Syntax::Methods

puts "Testing FactoryBot factories..."
puts "=" * 50

# Attempt to build a player
puts "\nBuilding player..."
player1 = build(:player)
puts "Player valid? #{player1.validate}"
puts "Player username: #{player1.username}"

# Attempt to build a game session
puts "\nBuilding game session..."
session1 = build(:game_session)
puts "Session valid? #{session1.validate}"
puts "Session ID: #{session1.session_id}"
puts "Session has player? #{!session1.player.nil?}"

# Attempt to build a score
puts "\nBuilding score..."
score1 = build(:score)
puts "Score valid? #{score1.validate}"
puts "Score points: #{score1.points}"
puts "Score has game session? #{!score1.game_session.nil?}"

puts "\n" + "=" * 50
puts "Factory test complete. Check validation results above."