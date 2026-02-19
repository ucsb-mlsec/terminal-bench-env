import { addClass, removeClass } from '@utils/domUtils';
import { validateProps } from '@utils/validators';

export class Button {
  constructor(props) {
    validateProps(props);
    this.label = props.label || 'Click me';
    this.disabled = props.disabled || false;
    this.element = null;
  }

  render() {
    this.element = document.createElement('button');
    this.element.textContent = this.label;
    addClass(this.element, 'btn');
    
    if (this.disabled) {
      this.element.disabled = true;
      addClass(this.element, 'btn-disabled');
    }
    
    return this.element;
  }

  enable() {
    this.disabled = false;
    this.element.disabled = false;
    removeClass(this.element, 'btn-disabled');
  }

  disable() {
    this.disabled = true;
    this.element.disabled = true;
    addClass(this.element, 'btn-disabled');
  }
}