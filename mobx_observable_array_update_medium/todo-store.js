const { makeObservable, observable, computed, action } = require('mobx');

class TodoStore {
  todos = [];

  constructor() {
    this.todos = ['Buy milk', 'Read book', 'Write code'];
    
    makeObservable(this, {
      todos: observable,
      itemCount: computed,
      addTodo: action
    });
  }

  itemCount() {
    return this.todos.length;
  }

  addTodo(todo) {
    this.todos.push(todo);
  }
}

module.exports = TodoStore;