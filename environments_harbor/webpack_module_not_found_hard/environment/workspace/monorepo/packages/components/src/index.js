import React from 'react';
import { formatText } from '@myorg/utils';

export const Button = (props) => {
  const formattedText = formatText(props.text || 'Click me');
  
  return (
    <button
      onClick={props.onClick}
      className={props.className}
      style={props.style}
    >
      {formattedText}
    </button>
  );
};

export default Button;