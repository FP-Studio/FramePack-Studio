# UI Templates

This directory contains HTML, CSS, and Markdown templates for the UI components. The template system uses Python's built-in `string.Template` class for safe variable substitution without requiring external dependencies.

## Template System Features

- **Zero Dependencies**: Uses only Python standard library (`string.Template`)
- **Safe Variable Substitution**: Protects against code injection
- **Template Caching**: Improves performance by caching loaded templates
- **Multiple File Types**: Supports HTML, CSS, and Markdown templates

## File Structure

```
modules/ui/templates/
├── README.md                # This documentation
├── queue_documentation.md   # Queue tab help documentation
├── thumbnail.html           # Thumbnail image template
├── queue_styles.css         # Queue-specific CSS styles
└── [your_custom_templates]  # Add your own templates here
```

## Usage Examples

### Loading Templates in Python

```python
from modules.ui.template_loader import template_loader, get_queue_documentation

# Load and render a template with variables
html_content = template_loader.render_html_template(
    'thumbnail.html', 
    thumbnail_url='path/to/image.jpg'
)

# Load markdown documentation
doc_content = get_queue_documentation()

# Load CSS styles
css_content = template_loader.load_css('queue_styles.css')
```

### Creating New Templates

1. **HTML Template** (`my_template.html`):
```html
<div class="$css_class">
  <h2>$title</h2>
  <p>$description</p>
  <img src="$image_url" alt="$alt_text">
</div>
```

2. **Using the Template**:
```python
content = template_loader.render_html_template(
    'my_template.html',
    css_class='my-component',
    title='My Title',
    description='Component description',
    image_url='/path/to/image.png',
    alt_text='Image description'
)
```

### Template Syntax

Templates use `$variable` syntax for substitution:

- `$variable` - Required variable (throws error if missing)
- `${variable}` - Required variable with explicit boundaries
- Template variables are case-sensitive

### Safe Substitution

The system uses `safe_substitute()` which:
- Leaves unmatched variables as-is (doesn't throw errors)
- Prevents code injection attacks
- Handles missing variables gracefully

## Best Practices

1. **Template Organization**:
   - Use descriptive filenames
   - Group related templates together
   - Keep templates focused on single components

2. **Variable Naming**:
   - Use clear, descriptive variable names
   - Follow snake_case convention
   - Document expected variables in comments

3. **Error Handling**:
   - Template loader gracefully handles missing files
   - Use try/catch when loading optional templates
   - Provide fallback content for missing variables

4. **Performance**:
   - Templates are automatically cached after first load
   - Avoid loading large templates repeatedly in loops
   - Consider template size for frequently used components

## Extending the System

### Adding New Template Types

```python
def render_my_template_type(template_name: str, **kwargs) -> str:
    """Render custom template type."""
    return template_loader.render_template(template_name, **kwargs)
```

### Custom Template Directories

```python
from modules.ui.template_loader import TemplateLoader

# Create loader for different directory
custom_loader = TemplateLoader('/path/to/custom/templates')
content = custom_loader.render_template('my_template.html', var='value')
```

## Migration Guide

When moving from inline HTML/CSS to templates:

1. **Extract Content**: Copy inline content to template files
2. **Identify Variables**: Replace dynamic content with `$variable` placeholders
3. **Update Code**: Replace inline content with template loader calls
4. **Test**: Verify templates render correctly with all variable combinations

## Template Examples

### Conditional Content
```html
<!-- In template file -->
<div class="status-$status">
  $content
</div>

<!-- In Python -->
template_loader.render_html_template(
    'status.html',
    status='success' if success else 'error',
    content='Operation completed' if success else 'Operation failed'
)
```

### Complex Variables
```html
<!-- Template supports complex variable names -->
<img src="$item_thumbnail_url" 
     width="$dimensions_width" 
     height="$dimensions_height">
```

This template system provides a clean, dependency-free way to separate presentation from logic while maintaining full compatibility with your existing Gradio application.