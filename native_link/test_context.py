import asyncio
import contextvars

var = contextvars.ContextVar('var', default='default')

async def inner():
    return var.get()

async def outer():
    var.set('outer')
    ctx = contextvars.Context()
    t = asyncio.create_task(inner(), context=ctx)
    return await t

print(asyncio.run(outer()))
