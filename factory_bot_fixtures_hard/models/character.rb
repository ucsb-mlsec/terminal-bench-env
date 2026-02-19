class Character
  attr_accessor :name, :level, :character_class, :health, :equipped_weapon, :equipped_armor

  VALID_CLASSES = ['warrior', 'mage', 'ranger', 'rogue', 'cleric'].freeze
  MIN_LEVEL = 1
  MAX_LEVEL = 100
  MIN_NAME_LENGTH = 3

  def initialize(attributes = {})
    @name = attributes[:name]
    @level = attributes[:level]
    @character_class = attributes[:character_class]
    @health = attributes[:health]
    @equipped_weapon = attributes[:equipped_weapon]
    @equipped_armor = attributes[:equipped_armor]
    @errors = []
  end

  def valid?
    @errors = []
    validate
    @errors.empty?
  end

  def errors
    @errors
  end

  private

  def validate
    validate_name
    validate_level
    validate_character_class
    validate_health
    validate_equipped_weapon
    validate_equipped_armor
  end

  def validate_name
    if @name.nil? || @name.to_s.strip.empty?
      @errors << "Name must be present"
      return
    end

    if @name.length < MIN_NAME_LENGTH
      @errors << "Name must be at least #{MIN_NAME_LENGTH} characters"
    end
  end

  def validate_level
    if @level.nil?
      @errors << "Level must be present"
      return
    end

    unless @level.is_a?(Integer)
      @errors << "Level must be an integer"
      return
    end

    if @level < MIN_LEVEL || @level > MAX_LEVEL
      @errors << "Level must be between #{MIN_LEVEL} and #{MAX_LEVEL}"
    end
  end

  def validate_character_class
    if @character_class.nil? || @character_class.to_s.strip.empty?
      @errors << "Character class must be present"
      return
    end

    unless VALID_CLASSES.include?(@character_class)
      @errors << "Character class must be one of: #{VALID_CLASSES.join(', ')}"
    end
  end

  def validate_health
    if @health.nil?
      @errors << "Health must be present"
      return
    end

    unless @health.is_a?(Numeric)
      @errors << "Health must be a number"
      return
    end

    if @health <= 0
      @errors << "Health must be positive"
    end
  end

  def validate_equipped_weapon
    return if @equipped_weapon.nil?

    unless @equipped_weapon.is_a?(Weapon)
      @errors << "Equipped weapon must be a Weapon instance"
      return
    end

    unless @equipped_weapon.valid?
      @errors << "Equipped weapon is invalid"
      return
    end

    if @level < @equipped_weapon.level_requirement
      @errors << "Character level (#{@level}) is too low for equipped weapon (requires level #{@equipped_weapon.level_requirement})"
    end
  end

  def validate_equipped_armor
    return if @equipped_armor.nil?

    unless @equipped_armor.is_a?(Armor)
      @errors << "Equipped armor must be an Armor instance"
      return
    end

    unless @equipped_armor.valid?
      @errors << "Equipped armor is invalid"
      return
    end

    if @level < @equipped_armor.level_requirement
      @errors << "Character level (#{@level}) is too low for equipped armor (requires level #{@equipped_armor.level_requirement})"
    end
  end
end