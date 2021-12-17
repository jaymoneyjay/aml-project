# aml-project
[Team Notes](https://docs.google.com/document/d/10n9X8kCpce0fTJFlTte9ZImwn17amqsj9S-B-TiHX2s/edit?usp=sharing)

## Development

### Logging

```python
from loguru import logger

# logging to levels
logger.trace()
logger.debug()
logger.info()
logger.success()
logger.warning()
logger.error()
logger.critical()

# example
logger.warning('Sample name "{}" invalid.', name)
```