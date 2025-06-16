const newTodoInput = document.getElementById('newTodo');
const addTodoButton = document.getElementById('addTodo');
const todoList = document.getElementById('todoList');

let todos = [];

addTodoButton.addEventListener('click', () => {
  const newTodoText = newTodoInput.value.trim();
  if (newTodoText !== '') {
    todos.push(newTodoText);
    renderTodos();
    newTodoInput.value = '';
  }
});

function renderTodos() {
  todoList.innerHTML = '';
  todos.forEach(todo => {
    const li = document.createElement('li');
    li.textContent = todo;
    todoList.appendChild(li);
  });
}
