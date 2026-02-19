class GameSession
  attr_accessor :session_id, :player

  def initialize(session_id, player)
    @session_id = session_id
    @player = player
  end

  def validate
    !@session_id.nil? && !@session_id.to_s.empty? && !@player.nil?
  end
end