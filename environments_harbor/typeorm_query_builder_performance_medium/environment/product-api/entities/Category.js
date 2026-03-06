const { Entity, PrimaryGeneratedColumn, Column } = require('typeorm');

@Entity()
class Category {
  @PrimaryGeneratedColumn()
  id;

  @Column()
  name;

  @Column({ type: 'text', nullable: true })
  description;
}

module.exports = Category;