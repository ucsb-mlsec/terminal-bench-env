class Player
  attr_accessor :username

  @@all_players = []

  def initialize(username)
    @username = username
    @@all_players << self
  end

  def validate
    return false if @username.nil? || @username.empty?
    
    # Check uniqueness
    @@all_players.count { |p| p.username == @username && p.object_id != self.object_id } == 0
  end

  def self.all
    @@all_players
  end

  def self.clear_all
    @@all_players = []
  end
end