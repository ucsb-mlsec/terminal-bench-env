class Score
  attr_accessor :points, :game_session

  def initialize(points, game_session)
    @points = points
    @game_session = game_session
  end

  def validate
    return false if @game_session.nil?
    return false if @points.nil?
    return false if @points < 0 || @points > 1000
    true
  end
end