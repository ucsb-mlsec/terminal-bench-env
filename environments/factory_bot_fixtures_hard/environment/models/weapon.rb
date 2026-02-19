# models/weapon.rb
class Weapon
  attr_accessor :name, :damage, :weapon_type, :rarity, :level_requirement

  VALID_WEAPON_TYPES = ['sword', 'axe', 'bow', 'staff', 'dagger', 'mace'].freeze
  VALID_RARITIES = ['common', 'uncommon', 'rare', 'epic', 'legendary'].freeze

  def initialize(attributes = {})
    @name = attributes[:name]
    @damage = attributes[:damage]
    @weapon_type = attributes[:weapon_type]
    @rarity = attributes[:rarity]
    @level_requirement = attributes[:level_requirement]
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
    validate_damage
    validate_weapon_type
    validate_rarity
    validate_level_requirement
  end

  def validate_name
    if @name.nil? || @name.to_s.strip.empty?
      @errors << "Name must be present"
    end
  end

  def validate_damage
    if @damage.nil?
      @errors << "Damage must be present"
    elsif !@damage.is_a?(Numeric)
      @errors << "Damage must be a number"
    elsif @damage <= 0
      @errors << "Damage must be greater than 0"
    end
  end

  def validate_weapon_type
    if @weapon_type.nil? || @weapon_type.to_s.strip.empty?
      @errors << "Weapon type must be present"
    elsif !VALID_WEAPON_TYPES.include?(@weapon_type.to_s)
      @errors << "Weapon type must be one of: #{VALID_WEAPON_TYPES.join(', ')}"
    end
  end

  def validate_rarity
    if @rarity.nil? || @rarity.to_s.strip.empty?
      @errors << "Rarity must be present"
    elsif !VALID_RARITIES.include?(@rarity.to_s)
      @errors << "Rarity must be one of: #{VALID_RARITIES.join(', ')}"
    end
  end

  def validate_level_requirement
    if @level_requirement.nil?
      @errors << "Level requirement must be present"
    elsif !@level_requirement.is_a?(Integer)
      @errors << "Level requirement must be an integer"
    elsif @level_requirement < 1 || @level_requirement > 100
      @errors << "Level requirement must be between 1 and 100"
    end
  end
end