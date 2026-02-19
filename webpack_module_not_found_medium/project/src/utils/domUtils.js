export function addClass(element, className) {
  if (element && className) {
    element.classList.add(className);
  }
}

export function removeClass(element, className) {
  if (element && className) {
    element.classList.remove(className);
  }
}

export function toggleClass(element, className) {
  if (element && className) {
    element.classList.toggle(className);
  }
}