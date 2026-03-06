const { Entity, PrimaryGeneratedColumn, Column, ManyToOne, JoinColumn } = require('typeorm');
const { Category } = require('./Category');

@Entity()
class Product {
  @PrimaryGeneratedColumn()
  id;

  @Column()
  name;

  @Column('decimal')
  price;

  @Column('text')
  description;

  @Column()
  categoryId;

  @ManyToOne(() => Category)
  @JoinColumn({ name: 'categoryId' })
  category;
}

module.exports = { Product };