class Armor
  attr_accessor :name, :defense, :armor_type, :rarity, :level_requirement, :slot

  VALID_ARMOR_TYPES = ['helmet', 'chest', 'legs', 'boots', 'gloves', 'shield'].freeze
  VALID_RARITIES = ['common', 'uncommon', 'rare', 'epic', 'legendary'].freeze
  MIN_LEVEL = 1
  MAX_LEVEL = 100

  def initialize(attributes = {})
    @name = attributes[:name]
    @defense = attributes[:defense]
    @armor_type = attributes[:armor_type]
    @rarity = attributes[:rarity]
    @level_requirement = attributes[:level_requirement]
    @slot = attributes[:slot]
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
    validate_defense
    validate_armor_type
    validate_rarity
    validate_level_requirement
    validate_slot
  end

  def validate_name
    if name.nil? || name.to_s.strip.empty?
      @errors << "Name must be present"
    end
  end

  def validate_defense
    if defense.nil?
      @errors << "Defense must be present"
    elsif !defense.is_a?(Numeric)
      @errors << "Defense must be a number"
    elsif defense <= 0
      @errors << "Defense must be greater than 0"
    end
  end

  def validate_armor_type
    if armor_type.nil? || armor_type.to_s.strip.empty?
      @errors << "Armor type must be present"
    elsif !VALID_ARMOR_TYPES.include?(armor_type.to_s)
      @errors << "Armor type must be one of: #{VALID_ARMOR_TYPES.join(', ')}"
    end
  end

  def validate_rarity
    if rarity.nil? || rarity.to_s.strip.empty?
      @errors << "Rarity must be present"
    elsif !VALID_RARITIES.include?(rarity.to_s)
      @errors << "Rarity must be one of: #{VALID_RARITIES.join(', ')}"
    end
  end

  def validate_level_requirement
    if level_requirement.nil?
      @errors << "Level requirement must be present"
    elsif !level_requirement.is_a?(Integer)
      @errors << "Level requirement must be an integer"
    elsif level_requirement < MIN_LEVEL || level_requirement > MAX_LEVEL
      @errors << "Level requirement must be between #{MIN_LEVEL} and #{MAX_LEVEL}"
    end
  end

  def validate_slot
    if slot.nil? || slot.to_s.strip.empty?
      @errors << "Slot must be present"
    elsif slot.to_s != armor_type.to_s
      @errors << "Slot must match armor type"
    end
  end
end